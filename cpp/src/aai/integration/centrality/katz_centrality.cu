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
#include "centrality/katz_centrality_impl.cuh"

#ifdef AAI_ROUTE_KATZ_CENTRALITY

#include <cugraph/aai/algorithms.hpp>
#include <cugraph/utilities/error.hpp>

#include <cuda_runtime.h>

namespace cugraph {

// =============================================================================
// Katz Centrality - AAI Specializations
// =============================================================================

template <>
void katz_centrality<int32_t, int32_t, float, float, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, true, false> const& graph_view,
    std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
    float const* betas,
    float* katz_centralities,
    float alpha,
    float beta,
    float epsilon,
    size_t max_iterations,
    bool has_initial_guess,
    bool normalize,
    bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (from katz_centrality_impl.cuh)
  CUGRAPH_EXPECTS(compact_graph.is_csc, "Katz centrality requires CSC format (is_csc=true)");
  CUGRAPH_EXPECTS((alpha >= 0.0f) && (alpha <= 1.0f),
                  "Invalid input argument: alpha should be in [0.0, 1.0].");
  CUGRAPH_EXPECTS(epsilon >= 0.0f, "Invalid input argument: epsilon should be non-negative.");

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  bool has_segments = compact_graph.segment_offsets.has_value();
  bool has_mask = compact_graph.edge_mask != nullptr;

  aai::katz_centrality_result_t result;
  if (edge_weight_view.has_value()) {
    const float* weights = edge_weight_view->value_firsts()[0];
    if (has_mask) {
      if (has_segments) {
        result = aai::katz_centrality_seg_mask(compact_graph, weights, katz_centralities,
                                       alpha, beta, betas, epsilon, max_iterations, has_initial_guess, normalize);
      } else {
        result = aai::katz_centrality_mask(compact_graph, weights, katz_centralities,
                                   alpha, beta, betas, epsilon, max_iterations, has_initial_guess, normalize);
      }
    } else {
      if (has_segments) {
        result = aai::katz_centrality_seg(compact_graph, weights, katz_centralities,
                                 alpha, beta, betas, epsilon, max_iterations, has_initial_guess, normalize);
      } else {
        result = aai::katz_centrality(compact_graph, weights, katz_centralities,
                             alpha, beta, betas, epsilon, max_iterations, has_initial_guess, normalize);
      }
    }
  } else {
    if (has_mask) {
      if (has_segments) {
        result = aai::katz_centrality_seg_mask(compact_graph, katz_centralities,
                                       alpha, beta, betas, epsilon, max_iterations, has_initial_guess, normalize);
      } else {
        result = aai::katz_centrality_mask(compact_graph, katz_centralities,
                                   alpha, beta, betas, epsilon, max_iterations, has_initial_guess, normalize);
      }
    } else {
      if (has_segments) {
        result = aai::katz_centrality_seg(compact_graph, katz_centralities,
                                 alpha, beta, betas, epsilon, max_iterations, has_initial_guess, normalize);
      } else {
        result = aai::katz_centrality(compact_graph, katz_centralities,
                             alpha, beta, betas, epsilon, max_iterations, has_initial_guess, normalize);
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
    CUGRAPH_FAIL("Katz Centrality failed to converge.");
  }
}

// Note: cuGraph's unweighted algorithms default weight_t to float32, so there is no
// unweighted double katz_centrality variant. Double precision requires explicit edge weights.
template <>
void katz_centrality<int32_t, int32_t, double, double, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, true, false> const& graph_view,
    std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
    double const* betas,
    double* katz_centralities,
    double alpha,
    double beta,
    double epsilon,
    size_t max_iterations,
    bool has_initial_guess,
    bool normalize,
    bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (from katz_centrality_impl.cuh)
  CUGRAPH_EXPECTS(compact_graph.is_csc, "Katz centrality requires CSC format (is_csc=true)");
  CUGRAPH_EXPECTS((alpha >= 0.0) && (alpha <= 1.0),
                  "Invalid input argument: alpha should be in [0.0, 1.0].");
  CUGRAPH_EXPECTS(epsilon >= 0.0, "Invalid input argument: epsilon should be non-negative.");
  CUGRAPH_EXPECTS(edge_weight_view.has_value(),
                  "Invalid input argument: double-precision Katz centrality requires edge weights.");

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  const double* weights = edge_weight_view->value_firsts()[0];

  // 4-way dispatch: mask x segment
  aai::katz_centrality_result_t result;
  if (compact_graph.edge_mask != nullptr) {
    if (compact_graph.segment_offsets.has_value()) {
      result = aai::katz_centrality_seg_mask(compact_graph, weights, katz_centralities,
                                     alpha, beta, betas, epsilon, max_iterations, has_initial_guess, normalize);
    } else {
      result = aai::katz_centrality_mask(compact_graph, weights, katz_centralities,
                                 alpha, beta, betas, epsilon, max_iterations, has_initial_guess, normalize);
    }
  } else {
    if (compact_graph.segment_offsets.has_value()) {
      result = aai::katz_centrality_seg(compact_graph, weights, katz_centralities,
                               alpha, beta, betas, epsilon, max_iterations, has_initial_guess, normalize);
    } else {
      result = aai::katz_centrality(compact_graph, weights, katz_centralities,
                           alpha, beta, betas, epsilon, max_iterations, has_initial_guess, normalize);
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
    CUGRAPH_FAIL("Katz Centrality failed to converge.");
  }
}

}  // namespace cugraph

#else  // !AAI_ROUTE_KATZ_CENTRALITY

namespace cugraph {

template void katz_centrality<int32_t, int32_t, float, float, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, true, false> const& graph_view,
    std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
    float const* betas,
    float* katz_centralities,
    float alpha,
    float beta,
    float epsilon,
    size_t max_iterations,
    bool has_initial_guess,
    bool normalize,
    bool do_expensive_check);

template void katz_centrality<int32_t, int32_t, double, double, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, true, false> const& graph_view,
    std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
    double const* betas,
    double* katz_centralities,
    double alpha,
    double beta,
    double epsilon,
    size_t max_iterations,
    bool has_initial_guess,
    bool normalize,
    bool do_expensive_check);

}  // namespace cugraph

#endif  // AAI_ROUTE_KATZ_CENTRALITY
