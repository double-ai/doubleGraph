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
#include "community/louvain_impl.cuh"

namespace cugraph {

// =============================================================================
// Original Template Instantiations (Dendrogram-returning versions)
// =============================================================================

template std::pair<std::unique_ptr<Dendrogram<int32_t>>, float> louvain(
  raft::handle_t const&,
  std::optional<std::reference_wrapper<raft::random::RngState>>,
  graph_view_t<int32_t, int32_t, false, false> const&,
  std::optional<edge_property_view_t<int32_t, float const*>>,
  size_t,
  float,
  float);

template std::pair<std::unique_ptr<Dendrogram<int32_t>>, double> louvain(
  raft::handle_t const&,
  std::optional<std::reference_wrapper<raft::random::RngState>>,
  graph_view_t<int32_t, int32_t, false, false> const&,
  std::optional<edge_property_view_t<int32_t, double const*>>,
  size_t,
  double,
  double);

}  // namespace cugraph

#ifdef AAI_ROUTE_LOUVAIN

#include <cugraph/aai/algorithms.hpp>
#include <cugraph/utilities/error.hpp>

#include <cuda_runtime.h>

#include <stdexcept>

namespace cugraph {

// =============================================================================
// AAI Specializations (clustering-array returning versions)
// =============================================================================

template <>
std::pair<size_t, float> louvain<int32_t, int32_t, float, false>(
    raft::handle_t const& handle,
    std::optional<std::reference_wrapper<raft::random::RngState>> rng_state,
    graph_view_t<int32_t, int32_t, false, false> const& graph_view,
    std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
    int32_t* clustering,
    size_t max_level,
    float threshold,
    float resolution)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (from louvain_impl.cuh)
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "Edge masks are not supported");
  CUGRAPH_EXPECTS(edge_weight_view.has_value(), "Graph must be weighted");
  CUGRAPH_EXPECTS(!compact_graph.is_csc, "Louvain requires CSR format (is_csc=false)");
  CUGRAPH_EXPECTS(clustering != nullptr, "clustering output pointer must not be null");

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  const float* weights = edge_weight_view->value_firsts()[0];
  aai::louvain_result_float_t result;

  if (compact_graph.segment_offsets.has_value()) {
    result = aai::louvain_seg(compact_graph, weights, clustering, max_level, threshold, resolution);
  } else {
    result = aai::louvain(compact_graph, weights, clustering, max_level, threshold, resolution);
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }
  return std::make_pair(result.level_count, result.modularity);
}

template <>
std::pair<size_t, double> louvain<int32_t, int32_t, double, false>(
    raft::handle_t const& handle,
    std::optional<std::reference_wrapper<raft::random::RngState>> rng_state,
    graph_view_t<int32_t, int32_t, false, false> const& graph_view,
    std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
    int32_t* clustering,
    size_t max_level,
    double threshold,
    double resolution)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (from louvain_impl.cuh)
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "Edge masks are not supported");
  CUGRAPH_EXPECTS(edge_weight_view.has_value(), "Graph must be weighted");
  CUGRAPH_EXPECTS(!compact_graph.is_csc, "Louvain requires CSR format (is_csc=false)");
  CUGRAPH_EXPECTS(clustering != nullptr, "clustering output pointer must not be null");

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  const double* weights = edge_weight_view->value_firsts()[0];
  aai::louvain_result_double_t result;

  if (compact_graph.segment_offsets.has_value()) {
    result = aai::louvain_seg(compact_graph, weights, clustering, max_level, threshold, resolution);
  } else {
    result = aai::louvain(compact_graph, weights, clustering, max_level, threshold, resolution);
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }
  return std::make_pair(result.level_count, result.modularity);
}

}  // namespace cugraph

#else  // !AAI_ROUTE_LOUVAIN

namespace cugraph {
template std::pair<size_t, float> louvain<int32_t, int32_t, float, false>(raft::handle_t const&, std::optional<std::reference_wrapper<raft::random::RngState>>, graph_view_t<int32_t, int32_t, false, false> const&, std::optional<edge_property_view_t<int32_t, float const*>>, int32_t*, size_t, float, float);
template std::pair<size_t, double> louvain<int32_t, int32_t, double, false>(raft::handle_t const&, std::optional<std::reference_wrapper<raft::random::RngState>>, graph_view_t<int32_t, int32_t, false, false> const&, std::optional<edge_property_view_t<int32_t, double const*>>, int32_t*, size_t, double, double);
}  // namespace cugraph

#endif  // AAI_ROUTE_LOUVAIN
