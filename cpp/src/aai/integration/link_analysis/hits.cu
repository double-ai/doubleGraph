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
#include "link_analysis/hits_impl.cuh"

#ifdef AAI_ROUTE_HITS

#include <cugraph/aai/algorithms.hpp>
#include <cugraph/utilities/error.hpp>

#include <cuda_runtime.h>

namespace cugraph {

// =============================================================================
// HITS - AAI Specialization (float only)
// =============================================================================

template <>
std::tuple<float, size_t> hits<int32_t, int32_t, float, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, true, false> const& graph_view,
    float* const hubs,
    float* const authorities,
    float epsilon,
    size_t max_iterations,
    bool has_initial_hubs_guess,
    bool normalize,
    bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (match original hits_impl.cuh)
  CUGRAPH_EXPECTS(compact_graph.is_csc, "HITS requires CSC format (is_csc=true)");
  CUGRAPH_EXPECTS(epsilon >= 0.0f, "Invalid input argument: epsilon should be non-negative.");
  CUGRAPH_EXPECTS(hubs != nullptr, "hubs output pointer must not be null");
  CUGRAPH_EXPECTS(authorities != nullptr, "authorities output pointer must not be null");

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  aai::HitsResult result;
  if (compact_graph.edge_mask != nullptr) {
    if (compact_graph.segment_offsets.has_value()) {
      result = aai::hits_seg_mask(compact_graph, hubs, authorities, epsilon, max_iterations, has_initial_hubs_guess, normalize);
    } else {
      result = aai::hits_mask(compact_graph, hubs, authorities, epsilon, max_iterations, has_initial_hubs_guess, normalize);
    }
  } else {
    if (compact_graph.segment_offsets.has_value()) {
      result = aai::hits_seg(compact_graph, hubs, authorities, epsilon, max_iterations, has_initial_hubs_guess, normalize);
    } else {
      result = aai::hits(compact_graph, hubs, authorities, epsilon, max_iterations, has_initial_hubs_guess, normalize);
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
    CUGRAPH_FAIL("HITS failed to converge.");
  }

  return std::make_tuple(result.final_norm, result.iterations);
}

template <>
std::tuple<double, size_t> hits<int32_t, int32_t, double, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, true, false> const& graph_view,
    double* const hubs,
    double* const authorities,
    double epsilon,
    size_t max_iterations,
    bool has_initial_hubs_guess,
    bool normalize,
    bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (match original hits_impl.cuh)
  CUGRAPH_EXPECTS(compact_graph.is_csc, "HITS requires CSC format (is_csc=true)");
  CUGRAPH_EXPECTS(epsilon >= 0.0, "Invalid input argument: epsilon should be non-negative.");
  CUGRAPH_EXPECTS(hubs != nullptr, "hubs output pointer must not be null");
  CUGRAPH_EXPECTS(authorities != nullptr, "authorities output pointer must not be null");

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  aai::HitsResultDouble result;
  if (compact_graph.edge_mask != nullptr) {
    if (compact_graph.segment_offsets.has_value()) {
      result = aai::hits_seg_mask(
        compact_graph, hubs, authorities, epsilon, max_iterations, has_initial_hubs_guess, normalize);
    } else {
      result = aai::hits_mask(
        compact_graph, hubs, authorities, epsilon, max_iterations, has_initial_hubs_guess, normalize);
    }
  } else {
    if (compact_graph.segment_offsets.has_value()) {
      result = aai::hits_seg(
        compact_graph, hubs, authorities, epsilon, max_iterations, has_initial_hubs_guess, normalize);
    } else {
      result = aai::hits(
        compact_graph, hubs, authorities, epsilon, max_iterations, has_initial_hubs_guess, normalize);
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
    CUGRAPH_FAIL("HITS failed to converge.");
  }

  return std::make_tuple(result.final_norm, result.iterations);
}

}  // namespace cugraph

#else

namespace cugraph {
template std::tuple<float, size_t> hits<int32_t, int32_t, float, false>(raft::handle_t const&, graph_view_t<int32_t, int32_t, true, false> const&, float* const, float* const, float, size_t, bool, bool, bool);
template std::tuple<double, size_t> hits<int32_t, int32_t, double, false>(raft::handle_t const&, graph_view_t<int32_t, int32_t, true, false> const&, double* const, double* const, double, size_t, bool, bool, bool);
}  // namespace cugraph

#endif
