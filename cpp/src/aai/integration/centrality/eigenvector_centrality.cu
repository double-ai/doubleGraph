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
#include "centrality/eigenvector_centrality_impl.cuh"

#ifdef AAI_ROUTE_EIGENVECTOR_CENTRALITY

#include <cugraph/aai/algorithms.hpp>
#include <cugraph/utilities/error.hpp>

#include <cuda_runtime.h>

namespace cugraph {

// =============================================================================
// Eigenvector Centrality - AAI Specializations
// =============================================================================

template <>
rmm::device_uvector<float> eigenvector_centrality<int32_t, int32_t, float, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, true, false> const& graph_view,
    std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
    std::optional<raft::device_span<float const>> initial_centralities,
    float epsilon,
    size_t max_iterations,
    bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (from eigenvector_centrality_impl.cuh)
  CUGRAPH_EXPECTS(compact_graph.is_csc, "Eigenvector centrality requires CSC format (is_csc=true)");
  CUGRAPH_EXPECTS(epsilon >= 0.0f, "Invalid input argument: epsilon should be non-negative.");
  if (initial_centralities)
    CUGRAPH_EXPECTS(initial_centralities->size() ==
                      static_cast<size_t>(graph_view.local_vertex_partition_range_size()),
                    "Centralities should be same size as vertex range");

  rmm::device_uvector<float> centralities(
      graph_view.local_vertex_partition_range_size(), handle.get_stream());

  const float* initial_ptr = initial_centralities ? initial_centralities->data() : nullptr;

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  bool has_segments = compact_graph.segment_offsets.has_value();
  bool has_mask = compact_graph.edge_mask != nullptr;

  aai::eigenvector_centrality_result_t result;
  if (edge_weight_view.has_value()) {
    const float* weights = edge_weight_view->value_firsts()[0];
    if (has_mask) {
      if (has_segments) {
        result = aai::eigenvector_centrality_seg_mask(compact_graph, weights, centralities.data(),
                                              epsilon, max_iterations, initial_ptr);
      } else {
        result = aai::eigenvector_centrality_mask(compact_graph, weights, centralities.data(),
                                          epsilon, max_iterations, initial_ptr);
      }
    } else {
      if (has_segments) {
        result = aai::eigenvector_centrality_seg(compact_graph, weights, centralities.data(),
                                        epsilon, max_iterations, initial_ptr);
      } else {
        result = aai::eigenvector_centrality(compact_graph, weights, centralities.data(),
                                    epsilon, max_iterations, initial_ptr);
      }
    }
  } else {
    if (has_mask) {
      if (has_segments) {
        result = aai::eigenvector_centrality_seg_mask(compact_graph, centralities.data(),
                                              epsilon, max_iterations, initial_ptr);
      } else {
        result = aai::eigenvector_centrality_mask(compact_graph, centralities.data(),
                                          epsilon, max_iterations, initial_ptr);
      }
    } else {
      if (has_segments) {
        result = aai::eigenvector_centrality_seg(compact_graph, centralities.data(),
                                        epsilon, max_iterations, initial_ptr);
      } else {
        result = aai::eigenvector_centrality(compact_graph, centralities.data(),
                                    epsilon, max_iterations, initial_ptr);
      }
    }
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }

  if (!result.converged) {
    CUGRAPH_FAIL("Eigenvector Centrality failed to converge.");
  }

  return centralities;
}

// Note: cuGraph's unweighted algorithms default weight_t to float32, so there is no
// unweighted double eigenvector_centrality variant. Double precision requires explicit edge weights.
template <>
rmm::device_uvector<double> eigenvector_centrality<int32_t, int32_t, double, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, true, false> const& graph_view,
    std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
    std::optional<raft::device_span<double const>> initial_centralities,
    double epsilon,
    size_t max_iterations,
    bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (from eigenvector_centrality_impl.cuh)
  CUGRAPH_EXPECTS(compact_graph.is_csc, "Eigenvector centrality requires CSC format (is_csc=true)");
  CUGRAPH_EXPECTS(epsilon >= 0.0, "Invalid input argument: epsilon should be non-negative.");
  if (initial_centralities)
    CUGRAPH_EXPECTS(initial_centralities->size() ==
                      static_cast<size_t>(graph_view.local_vertex_partition_range_size()),
                    "Centralities should be same size as vertex range");
  CUGRAPH_EXPECTS(edge_weight_view.has_value(),
                  "Invalid input argument: double-precision eigenvector centrality requires edge weights.");

  rmm::device_uvector<double> centralities(
      graph_view.local_vertex_partition_range_size(), handle.get_stream());

  const double* initial_ptr = initial_centralities ? initial_centralities->data() : nullptr;

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  const double* weights = edge_weight_view->value_firsts()[0];

  // 4-way dispatch: mask x segment
  aai::eigenvector_centrality_result_t result;
  if (compact_graph.edge_mask != nullptr) {
    if (compact_graph.segment_offsets.has_value()) {
      result = aai::eigenvector_centrality_seg_mask(compact_graph, weights, centralities.data(),
                                            epsilon, max_iterations, initial_ptr);
    } else {
      result = aai::eigenvector_centrality_mask(compact_graph, weights, centralities.data(),
                                        epsilon, max_iterations, initial_ptr);
    }
  } else {
    if (compact_graph.segment_offsets.has_value()) {
      result = aai::eigenvector_centrality_seg(compact_graph, weights, centralities.data(),
                                      epsilon, max_iterations, initial_ptr);
    } else {
      result = aai::eigenvector_centrality(compact_graph, weights, centralities.data(),
                                  epsilon, max_iterations, initial_ptr);
    }
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }

  if (!result.converged) {
    CUGRAPH_FAIL("Eigenvector Centrality failed to converge.");
  }

  return centralities;
}

}  // namespace cugraph

#else  // !AAI_ROUTE_EIGENVECTOR_CENTRALITY

namespace cugraph {

template rmm::device_uvector<float> eigenvector_centrality<int32_t, int32_t, float, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, true, false> const& graph_view,
    std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
    std::optional<raft::device_span<float const>> initial_centralities,
    float epsilon,
    size_t max_iterations,
    bool do_expensive_check);

template rmm::device_uvector<double> eigenvector_centrality<int32_t, int32_t, double, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, true, false> const& graph_view,
    std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
    std::optional<raft::device_span<double const>> initial_centralities,
    double epsilon,
    size_t max_iterations,
    bool do_expensive_check);

}  // namespace cugraph

#endif  // AAI_ROUTE_EIGENVECTOR_CENTRALITY
