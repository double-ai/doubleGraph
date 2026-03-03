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
#include "community/ecg_impl.cuh"

#ifdef AAI_ROUTE_ECG

#include <cugraph/aai/algorithms.hpp>
#include <cugraph/utilities/error.hpp>

#include <cuda_runtime.h>

#include <stdexcept>

namespace cugraph {

// =============================================================================
// AAI Specializations for ECG
// =============================================================================

// Float specialization
template <>
std::tuple<rmm::device_uvector<int32_t>, size_t, float> ecg<int32_t, int32_t, float, false>(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  float min_weight,
  size_t ensemble_size,
  size_t max_level,
  float threshold,
  float resolution)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (from ecg_impl.cuh)
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "Edge masks are not supported");
  CUGRAPH_EXPECTS(!compact_graph.is_csc, "ECG requires CSR format (is_csc=false)");
  CUGRAPH_EXPECTS(edge_weight_view.has_value(), "Graph must be weighted");
  CUGRAPH_EXPECTS(min_weight >= float{0.0},
                  "Invalid input arguments: min_weight must be positive");
  CUGRAPH_EXPECTS(ensemble_size >= 1,
                  "Invalid input arguments: ensemble_size must be a non-zero integer");
  CUGRAPH_EXPECTS(threshold > 0.0f && threshold <= 1.0f,
                  "Invalid input arguments: threshold must be a positive number in range (0.0, 1.0]");
  CUGRAPH_EXPECTS(resolution > 0.0f && resolution <= 1.0f,
                  "Invalid input arguments: resolution must be a positive number in range (0.0, 1.0]");

  // Allocate output
  rmm::device_uvector<int32_t> clusters(graph_view.number_of_vertices(), handle.get_stream());

  // Note: rng_state is intentionally not used. AAI implementations are deterministic.

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  const float* weights = edge_weight_view->value_firsts()[0];
  aai::ecg_result_float_t result;

  if (compact_graph.segment_offsets.has_value()) {
    result = aai::ecg_seg(compact_graph, weights, clusters.data(),
                          min_weight, ensemble_size, max_level, threshold, resolution);
  } else {
    result = aai::ecg(compact_graph, weights, clusters.data(),
                      min_weight, ensemble_size, max_level, threshold, resolution);
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }

  return std::make_tuple(std::move(clusters), result.level_count, result.modularity);
}

// Double specialization
template <>
std::tuple<rmm::device_uvector<int32_t>, size_t, double> ecg<int32_t, int32_t, double, false>(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
  double min_weight,
  size_t ensemble_size,
  size_t max_level,
  double threshold,
  double resolution)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (from ecg_impl.cuh)
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "Edge masks are not supported");
  CUGRAPH_EXPECTS(!compact_graph.is_csc, "ECG requires CSR format (is_csc=false)");
  CUGRAPH_EXPECTS(edge_weight_view.has_value(), "Graph must be weighted");
  CUGRAPH_EXPECTS(min_weight >= double{0.0},
                  "Invalid input arguments: min_weight must be positive");
  CUGRAPH_EXPECTS(ensemble_size >= 1,
                  "Invalid input arguments: ensemble_size must be a non-zero integer");
  CUGRAPH_EXPECTS(threshold > 0.0 && threshold <= 1.0,
                  "Invalid input arguments: threshold must be a positive number in range (0.0, 1.0]");
  CUGRAPH_EXPECTS(resolution > 0.0 && resolution <= 1.0,
                  "Invalid input arguments: resolution must be a positive number in range (0.0, 1.0]");

  // Allocate output
  rmm::device_uvector<int32_t> clusters(graph_view.number_of_vertices(), handle.get_stream());

  // Note: rng_state is intentionally not used. AAI implementations are deterministic.

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  const double* weights = edge_weight_view->value_firsts()[0];
  aai::ecg_result_double_t result;

  if (compact_graph.segment_offsets.has_value()) {
    result = aai::ecg_seg(compact_graph, weights, clusters.data(),
                          min_weight, ensemble_size, max_level, threshold, resolution);
  } else {
    result = aai::ecg(compact_graph, weights, clusters.data(),
                      min_weight, ensemble_size, max_level, threshold, resolution);
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }

  return std::make_tuple(std::move(clusters), result.level_count, result.modularity);
}

}  // namespace cugraph

#else  // !AAI_ROUTE_ECG

namespace cugraph {
template std::tuple<rmm::device_uvector<int32_t>, size_t, float> ecg<int32_t, int32_t, float, false>(raft::handle_t const&, raft::random::RngState&, graph_view_t<int32_t, int32_t, false, false> const&, std::optional<edge_property_view_t<int32_t, float const*>>, float, size_t, size_t, float, float);
template std::tuple<rmm::device_uvector<int32_t>, size_t, double> ecg<int32_t, int32_t, double, false>(raft::handle_t const&, raft::random::RngState&, graph_view_t<int32_t, int32_t, false, false> const&, std::optional<edge_property_view_t<int32_t, double const*>>, double, size_t, size_t, double, double);
}  // namespace cugraph

#endif  // AAI_ROUTE_ECG
